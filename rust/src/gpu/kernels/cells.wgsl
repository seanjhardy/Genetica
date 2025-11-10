// Compute shader for cell updates
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
}


@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(0) @binding(1)
var<storage, read_write> cells: array<Cell>;

struct CellFreeList {
    count: atomic<u32>,
    indices: array<u32>,
}

struct CellEvents {
    count: atomic<u32>,
    indices: array<u32>,
}

@group(0) @binding(2)
var<storage, read_write> cell_free_list: CellFreeList;

@group(0) @binding(3)
var<storage, read_write> cell_events: CellEvents;


fn rand(seed: vec2<u32>) -> f32 {
    var x = seed.x * 1664525u + 1013904223u;
    var y = seed.y * 22695477u + 1u;
    let n = x ^ y;
    return f32(n & 0x00FFFFFFu) / f32(0x01000000u) * 2.0 - 1.0;
}


@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;

    // Check if this cell index is in the free list first
    let free_cells_count = atomicLoad(&cell_free_list.count);
    for (var i: u32 = 0u; i < free_cells_count; i++) {
        if cell_free_list.indices[i] == index {
            return;
        }
    }

    var cell = cells[index];
    
    // Decrease energy over time (metabolic rate)
    let energy_decay_rate = 0.05 * cell.radius; // Energy units per second (slower decay)
    cell.energy -= energy_decay_rate * uniforms.sim_params.x;
    
    // Remove cell if energy reaches 0
    // TEMPORARILY DISABLED - don't remove cells based on energy
    if cell.energy <= 0.0 {
        cell.energy = 0.0;
        cells[index] = cell;
        
        // Add this cell's index to the free list atomically
        // Write the index at the next available slot using atomic operation
        let next_free_index = atomicAdd(&cell_free_list.count, 1u);
        cell_free_list.indices[next_free_index] = index;

        let event_index = atomicAdd(&cell_events.count, 1u);
        cell_events.indices[event_index] = index;
        
        // Update lifeform cell count (atomic operation not available in WGSL,
        // so we'll handle this on CPU side after reading back results)
        return;
    }
    
    // Generate random position offset for this timestep
    // Use cell index + current position as seed - this ensures each cell gets different randomness
    // and the randomness changes as the cell moves, creating more varied movement
    let pos_hash_x = u32(abs(cell.pos.x) * 7919.0) % 100000u;
    let pos_hash_y = u32(abs(cell.pos.y) * 1669.0) % 100000u;
    
    // Also use previous position to add more variation
    let prev_hash_x = u32(abs(cell.prev_pos.x) * 3319.0) % 100000u;
    let prev_hash_y = u32(abs(cell.prev_pos.y) * 4217.0) % 100000u;
    
    // Combine index, current position, and previous position for seed
    // Using different prime multipliers to ensure good distribution
    let seed1 = (index * 7919u + pos_hash_x * 17u + prev_hash_x * 31u) % 100000u;
    let seed2 = (index * 1669u + pos_hash_y * 23u + prev_hash_y * 37u) % 100000u;
    
    // Generate two independent random values
    let random_x = rand(vec2<u32>(seed1, seed2));
    let random_y = rand(vec2<u32>(seed2, seed1 * 3u + 41u));
    
    // Random position offset per timestep (added directly to position, no accumulation)
    let random_offset_magnitude = 0.5; // World units per timestep (small offset for subtle movement)
    let random_offset = vec2<f32>(random_x, random_y) * random_offset_magnitude * uniforms.sim_params.x;
    
    // Store random offset for potential future use (but not using it for accumulation anymore)
    cell.random_force = random_offset;
    
    // Verlet integration with damping (no acceleration term)
    let dt = uniforms.sim_params.x;
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
    
    cells[index] = cell;
}
