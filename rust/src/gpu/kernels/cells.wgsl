// Compute shader for cell updates
struct Cell {
    pos: vec2<f32>,
    prev_pos: vec2<f32>,
    random_force: vec2<f32>,
    radius: f32,
    energy: f32,
    cell_wall_thickness: f32,
    is_alive: u32,
    lifeform_slot: u32,
    metadata: u32,
}

struct Link {
    a: u32,
    b: u32,
    flags: u32,
    _padding0: u32,
    rest_length: f32,
    stiffness: f32,
    energy_transfer_rate: f32,
    _padding1: f32,
}

struct CellEvent {
    kind: u32,
    parent_cell_index: u32,
    parent_lifeform_slot: u32,
    flags: u32,
    position: vec2<f32>,
    radius: f32,
    energy: f32,
}

struct LinkEvent {
    kind: u32,
    link_index: u32,
    cell_a: u32,
    cell_b: u32,
    rest_length: f32,
    stiffness: f32,
    energy_transfer_rate: f32,
    _padding: f32,
}

// Uniforms struct must match Rust struct layout exactly (including padding)
struct Uniforms {
    sim_params: vec4<f32>, // x: dt, y: zoom, z: view_w, w: view_h
    cell_count: vec4<f32>, // x: cell_count, y: reserved0, z: reserved1, w: reserved2
    camera: vec4<f32>,     // x: cam_x, y: cam_y
    bounds: vec4<f32>,     // (left, top, right, bottom)
    nutrient: vec4<u32>,// (Cell size, scale, reserved, reserved)
}


@group(0) @binding(0)
var<uniform> uniforms: Uniforms;


@group(0) @binding(1)
var<storage, read_write> cells: array<Cell>;

struct CellFreeList {
    count: atomic<u32>,
    indices: array<u32>,
}

@group(0) @binding(2)
var<storage, read_write> cell_free_list: CellFreeList;

struct Counter {
    value: atomic<u32>,
}

@group(0) @binding(3)
var<storage, read_write> alive_counter: Counter;

@group(0) @binding(4)
var<storage, read_write> spawn_count: Counter;

@group(0) @binding(5)
var<storage, read> spawn_requests: array<Cell>;

struct LifeformFlagArray {
    values: array<atomic<u32>>,
}

@group(0) @binding(6)
var<storage, read_write> lifeform_active: LifeformFlagArray;

struct DivisionRequest {
    parent_lifeform_slot: u32,
    cell_index: u32,
    pos: vec2<f32>,
    radius: f32,
    energy: f32,
}

@group(0) @binding(7)
var<storage, read_write> division_request_count: Counter;

@group(0) @binding(8)
var<storage, read_write> division_requests: array<DivisionRequest>;

struct NutrientGrid {
    values: array<atomic<u32>>,
}

@group(0) @binding(9)
var<storage, read_write> nutrient_grid: NutrientGrid;

@group(0) @binding(10)
var<storage, read_write> links: array<Link>;

@group(0) @binding(11)
var<storage, read_write> link_free_count: Counter;

@group(0) @binding(12)
var<storage, read_write> link_free_list: array<u32>;

@group(0) @binding(13)
var<storage, read_write> link_event_count: Counter;

@group(0) @binding(14)
var<storage, read_write> link_events: array<LinkEvent>;

@group(0) @binding(15)
var<storage, read_write> cell_event_count: Counter;

@group(0) @binding(16)
var<storage, read_write> cell_events: array<CellEvent>;

const DIVISION_PROBABILITY: f32 = 0.0001;
const RANDOM_DEATH_PROBABILITY: f32 = 0.00005;
const MAX_DIVISION_REQUESTS: u32 = 512u;
const LIFEFORM_CAPACITY: u32 = 4096u;
const MIN_DIVISION_ENERGY: f32 = 20.0;

const CELL_EVENT_KIND_DIVISION: u32 = 1u;
const CELL_EVENT_KIND_DEATH: u32 = 2u;
const CELL_EVENT_FLAG_ADHESIVE: u32 = 1u;

const LINK_EVENT_KIND_CREATE: u32 = 1u;
const LINK_EVENT_KIND_REMOVE: u32 = 2u;

const LINK_FLAG_ALIVE: u32 = 1u;
const LINK_FLAG_ADHESIVE: u32 = 1u << 1u;


fn rand(seed: vec2<u32>) -> f32 {
    var x = seed.x * 1664525u + 1013904223u;
    var y = seed.y * 22695477u + 1u;
    let n = x ^ y;
    return f32(n & 0x00FFFFFFu) / f32(0x01000000u);
}

fn push_cell_event(
    kind: u32,
    parent_cell_index: u32,
    parent_lifeform_slot: u32,
    flags: u32,
    position: vec2<f32>,
    radius: f32,
    energy: f32,
) {
    let event_index = atomicAdd(&cell_event_count.value, 1u);
    if event_index < arrayLength(&cell_events) {
        cell_events[event_index] = CellEvent(
            kind,
            parent_cell_index,
            parent_lifeform_slot,
            flags,
            position,
            radius,
            energy,
        );
    } else {
        atomicSub(&cell_event_count.value, 1u);
    }
}


@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let dt = uniforms.sim_params.x;

    let random = get_random_values(index);

    spawn_cells();

    let total_cells = arrayLength(&cells);
    if index >= total_cells {
        return;
    }

    var cell = cells[index];
    if cell.is_alive == 0u {
        return;
    }

    // Decrease energy over time (metabolic rate)
    var energy_change_rate = 0.0;
    energy_change_rate -= 0.2 + 0.3 / cell.radius; // Metabolism proportional to size
    energy_change_rate += 1000.0 * absorb_nutrients(index, 0.001 * cell.radius); // Eat nutrients from the environemnt
    cell.energy += energy_change_rate * dt;
    cell.energy = clamp(cell.energy, 0.0, cell.radius * 100.0);

    if cell.energy <= 0.0 || random.z < RANDOM_DEATH_PROBABILITY {
        kill_cell(index);
        return;
    }
    
    // Random position offset per timestep (added directly to position, no accumulation)
    let random_offset_magnitude = 0.5; // World units per timestep (small offset for subtle movement)
    let random_offset = (random.xy * 2.0 - 1.0) * random_offset_magnitude * dt / min(cell.radius, 10.0);
    
    // Store random offset for potential future use (but not using it for accumulation anymore)
    cell.random_force = random_offset;
    
    // Verlet integration with damping (no acceleration term)
    let velocity = cell.pos - cell.prev_pos;
    
    let damping = 0.98;
    // Add random offset directly to position instead of using acceleration
    var new_pos = cell.pos + velocity * damping + random_offset;

    cell.prev_pos = cell.pos;
    cell.pos = new_pos;
    
    // Boundary constraints
    // Note: bounds is [left, top, right, bottom]
    let radius = cell.radius;
    let min_x = uniforms.bounds.x + radius;
    let max_x = uniforms.bounds.z - radius; // bounds.z is right edge
    let min_y = uniforms.bounds.y + radius;
    let max_y = uniforms.bounds.w - radius; // bounds.w is bottom edge
    
    if cell.pos.x < min_x {
        cell.prev_pos.x = cell.pos.x;
        cell.pos.x = min_x;
    } else if cell.pos.x > max_x {
        cell.prev_pos.x = cell.pos.x;
        cell.pos.x = max_x;
    }
    
    if cell.pos.y < min_y {
        cell.prev_pos.y = cell.pos.y;
        cell.pos.y = min_y;
    } else if cell.pos.y > max_y {
        cell.prev_pos.y = cell.pos.y;
        cell.pos.y = max_y;
    }


    if cell.energy > MIN_DIVISION_ENERGY {
        if random.w < DIVISION_PROBABILITY && cell.lifeform_slot < LIFEFORM_CAPACITY {
            let original_energy = cell.energy;
            let child_energy = original_energy * 0.5;
            let request_index = atomicAdd(&division_request_count.value, 1u);
            if request_index < MAX_DIVISION_REQUESTS {
                cell.energy = child_energy;
                division_requests[request_index].parent_lifeform_slot = cell.lifeform_slot;
                division_requests[request_index].cell_index = index;
                division_requests[request_index].pos = cell.pos;
                division_requests[request_index].radius = cell.radius;
                division_requests[request_index].energy = child_energy;
                push_cell_event(
                    CELL_EVENT_KIND_DIVISION,
                    index,
                    cell.lifeform_slot,
                    CELL_EVENT_FLAG_ADHESIVE,
                    cell.pos,
                    cell.radius,
                    child_energy,
                );
            } else {
                // Restore energy if we couldn't record the division
                atomicSub(&division_request_count.value, 1u);
                // No request recorded, keep original energy
                // (division didn't happen)
                cell.energy = original_energy;
            }
        }
    }

    cells[index] = cell;
}


fn spawn_cells() {
    loop {
        let prev_requests = atomicLoad(&spawn_count.value);
        if prev_requests == 0u {
            break;
        }
        let desired_requests = prev_requests - 1u;
        let request_exchange = atomicCompareExchangeWeak(
            &spawn_count.value,
            prev_requests,
            desired_requests,
        );
        if request_exchange.old_value == prev_requests && request_exchange.exchanged {
            let spawn_idx = desired_requests;
            var new_cell = spawn_requests[spawn_idx];
            let parent_marker = new_cell.metadata;
            new_cell.metadata = 0u;
            var spawned = false;
            loop {
                let free_prev = atomicLoad(&cell_free_list.count);
                if free_prev == 0u {
                    atomicAdd(&spawn_count.value, 1u);
                    break;
                }
                let free_desired = free_prev - 1u;
                let free_exchange = atomicCompareExchangeWeak(
                    &cell_free_list.count,
                    free_prev,
                    free_desired,
                );
                if free_exchange.old_value == free_prev && free_exchange.exchanged {
                    let slot_index = cell_free_list.indices[free_desired];
                    new_cell.is_alive = 1u;
                    cells[slot_index] = new_cell;
                    atomicAdd(&alive_counter.value, 1u);
                    let lf_idx = new_cell.lifeform_slot;
                    if lf_idx < LIFEFORM_CAPACITY {
                        atomicStore(&lifeform_active.values[lf_idx], 1u);
                    }
                    if parent_marker != 0u {
                        let parent_index = parent_marker - 1u;
                        if parent_index < arrayLength(&cells) {
                            var link_created = false;
                            loop {
                                let link_prev = atomicLoad(&link_free_count.value);
                                if link_prev == 0u {
                                    break;
                                }
                                let link_desired = link_prev - 1u;
                                let link_exchange = atomicCompareExchangeWeak(
                                    &link_free_count.value,
                                    link_prev,
                                    link_desired,
                                );
                                if link_exchange.old_value == link_prev && link_exchange.exchanged {
                                    let link_slot = link_free_list[link_desired];
                                    let parent_cell = cells[parent_index];
                                    let rest_length =
                                        parent_cell.radius + new_cell.radius;
                                    links[link_slot].a = parent_index;
                                    links[link_slot].b = slot_index;
                                    links[link_slot].flags =
                                        LINK_FLAG_ALIVE | LINK_FLAG_ADHESIVE;
                                    links[link_slot].rest_length = rest_length;
                                    links[link_slot].stiffness = 0.6;
                                    links[link_slot].energy_transfer_rate = 0.0;
                                    links[link_slot]._padding1 = 0.0;
                                    link_created = true;
                                    break;
                                }
                            }
                            if !link_created {
                                let event_index =
                                    atomicAdd(&link_event_count.value, 1u);
                                if event_index < arrayLength(&link_events) {
                                    link_events[event_index] = LinkEvent(
                                        LINK_EVENT_KIND_CREATE,
                                        0u,
                                        parent_index,
                                        slot_index,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                    );
                                } else {
                                    atomicSub(&link_event_count.value, 1u);
                                }
                            }
                        }
                    }
                    spawned = true;
                    break;
                }
            }
            if !spawned {
                break;
            }
        }
    }
}

fn kill_cell(index: u32) {
    var cell = cells[index];
    push_cell_event(
        CELL_EVENT_KIND_DEATH,
        index,
        cell.lifeform_slot,
        0u,
        cell.pos,
        cell.radius,
        cell.energy,
    );
    cell.energy = 0.0;
    cell.is_alive = 0u;
    cells[index] = cell;

    let next_free_index = atomicAdd(&cell_free_list.count, 1u);
    cell_free_list.indices[next_free_index] = index;

    loop {
        let current = atomicLoad(&alive_counter.value);
        if current == 0u {
            break;
        }
        let exchange = atomicCompareExchangeWeak(&alive_counter.value, current, current - 1u);
        if exchange.old_value == current && exchange.exchanged {
            break;
        }
    }

    let lf_idx = cell.lifeform_slot;
    if lf_idx < LIFEFORM_CAPACITY {
        atomicStore(&lifeform_active.values[lf_idx], 0u);
    }
}

fn get_random_values(index: u32) -> vec4<f32> {
    let cell = cells[index];
    let seed_1 = u32(abs(cell.pos.x + cell.energy) * 1669.0) % 100000u;
    let seed_2 = u32(abs(cell.pos.y + cell.radius) * 7919.0) % 100000u;

    let random_1 = rand(vec2<u32>(seed_1, seed_2));
    let random_2 = rand(vec2<u32>(seed_2, seed_1 * 3u + 41u));
    let random_3 = rand(vec2<u32>(seed_1 * 7u + 19u, seed_2 * 11u + 23u));
    let random_4 = rand(vec2<u32>(seed_2 * 13u + 29u, seed_1 * 17u + 31u));

    return vec4<f32>(random_1, random_2, random_3, random_4);
}

fn absorb_nutrients(index: u32, absorption_rate: f32) -> f32 {
    let cell = cells[index];
    let bounds_width = uniforms.bounds.z - uniforms.bounds.x;
    let bounds_height = uniforms.bounds.w - uniforms.bounds.y;
    let cell_size = f32(uniforms.nutrient.x);
    if bounds_width > 0.0 && bounds_height > 0.0 {
        let grid_width_f = max(1.0, ceil(bounds_width / cell_size));
        let grid_height_f = max(1.0, ceil(bounds_height / cell_size));
        let grid_width = u32(grid_width_f);
        let grid_height = u32(grid_height_f);

        let local_x = clamp(cell.pos.x - uniforms.bounds.x, 0.0, bounds_width - 0.0001);
        let local_y = clamp(cell.pos.y - uniforms.bounds.y, 0.0, bounds_height - 0.0001);

        let gx = u32(clamp(floor(local_x / cell_size), 0.0, grid_width_f - 1.0));
        let gy = u32(clamp(floor(local_y / cell_size), 0.0, grid_height_f - 1.0));
        let grid_index = gy * grid_width + gx;

        if grid_index < grid_width * grid_height && grid_index < arrayLength(&nutrient_grid.values) {
            var attempts = 0u;
            loop {
                let old_val = atomicLoad(&nutrient_grid.values[grid_index]);
                let current = f32(old_val) / f32(uniforms.nutrient.y);
                if current == 0.0 {
                    return 0.0;
                }

                let available = min(f32(absorption_rate), current);
                let new_val = u32(f32(old_val) - available * f32(uniforms.nutrient.y));

                let exchange = atomicCompareExchangeWeak(
                    &nutrient_grid.values[grid_index],
                    old_val,
                    new_val,
                );

                if exchange.exchanged {
                    return available;
                }

                attempts += 1u;
                if attempts > 4u {
                    break;
                }
            }
        }
    }
    return 0.0;
}
