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
    padding: u32,
}

// Uniforms struct must match Rust struct layout exactly (including padding)
struct Uniforms {
    sim_params: vec4<f32>, // x: dt, y: zoom, z: view_w, w: view_h
    cell_count: vec4<f32>, // x: cell_count, y: reserved0, z: reserved1, w: reserved2
    camera: vec4<f32>,     // x: cam_x, y: cam_y
    bounds: vec4<f32>,     // (left, top, right, bottom)
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

struct LifeformCounterArray {
    values: array<atomic<u32>>,
}

@group(0) @binding(6)
var<storage, read_write> lifeform_counts: LifeformCounterArray;

@group(0) @binding(7)
var<storage, read_write> lifeform_active: LifeformCounterArray;

@group(0) @binding(8)
var<storage, read_write> active_lifeform_counter: Counter;

struct DivisionRequest {
    parent_lifeform_slot: u32,
    cell_index: u32,
    pos: vec2<f32>,
    radius: f32,
    energy: f32,
}

@group(0) @binding(9)
var<storage, read_write> division_request_count: Counter;

@group(0) @binding(10)
var<storage, read_write> division_requests: array<DivisionRequest>;

const DIVISION_PROBABILITY: f32 = 0.01;
const RANDOM_DEATH_PROBABILITY: f32 = 0.00005;
const MAX_DIVISION_REQUESTS: u32 = 512u;
const LIFEFORM_CAPACITY: u32 = 4096u;
const MIN_DIVISION_ENERGY: f32 = 20.0;


fn rand(seed: vec2<u32>) -> f32 {
    var x = seed.x * 1664525u + 1013904223u;
    var y = seed.y * 22695477u + 1u;
    let n = x ^ y;
    return f32(n & 0x00FFFFFFu) / f32(0x01000000u);
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
    var energy_change_rate = 1.0 - 2.0 / cell.radius; // Energy units per second (slower decay)
    cell.energy += energy_change_rate * dt;

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
            let request_index = atomicAdd(&division_request_count.value, 1u);
            if request_index < MAX_DIVISION_REQUESTS {
                let child_energy = cell.energy * 0.5;
                cell.energy = child_energy;
                division_requests[request_index].parent_lifeform_slot = cell.lifeform_slot;
                division_requests[request_index].cell_index = index;
                division_requests[request_index].pos = cell.pos;
                division_requests[request_index].radius = cell.radius;
                division_requests[request_index].energy = child_energy;
            } else {
                atomicSub(&division_request_count.value, 1u);
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
                        let prev_lifeform = atomicAdd(&lifeform_counts.values[lf_idx], 1u);
                        if prev_lifeform == 0u {
                            let was_active = atomicExchange(&lifeform_active.values[lf_idx], 1u);
                            if was_active == 0u {
                                atomicAdd(&active_lifeform_counter.value, 1u);
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
        var lifeform_reached_zero = false;
        loop {
            let current = atomicLoad(&lifeform_counts.values[lf_idx]);
            if current == 0u {
                break;
            }
            let desired = current - 1u;
            let exchange = atomicCompareExchangeWeak(&lifeform_counts.values[lf_idx], current, desired);
            if exchange.old_value == current && exchange.exchanged {
                lifeform_reached_zero = desired == 0u;
                break;
            }
        }

        if lifeform_reached_zero {
            let was_active = atomicExchange(&lifeform_active.values[lf_idx], 0u);
            if was_active == 1u {
                loop {
                    let current = atomicLoad(&active_lifeform_counter.value);
                    if current == 0u {
                        break;
                    }
                    let exchange = atomicCompareExchangeWeak(&active_lifeform_counter.value, current, current - 1u);
                    if exchange.old_value == current && exchange.exchanged {
                        break;
                    }
                }
            }
        }
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
