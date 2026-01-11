@include src/gpu/wgsl/types.wgsl;
@include src/gpu/wgsl/constants.wgsl;
@include src/gpu/wgsl/utils/random.wgsl;

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> points: array<VerletPoint>;
@group(0) @binding(2) var<storage, read_write> points_counter: atomic<u32>;
@group(0) @binding(3) var<storage, read_write> points_free_list: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> cells: array<Cell>;
@group(0) @binding(5) var<storage, read_write> cells_counter: atomic<u32>;
@group(0) @binding(6) var<storage, read_write> cells_free_list: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> event_buffer: array<Event>;
@group(0) @binding(8) var<storage, read_write> event_counter: atomic<u32>;
@group(0) @binding(9) var<storage, read_write> lifeform_id: atomic<u32>;

// Check if we should spawn more cells (below capacity limit)
fn should_spawn_more(current_count: u32) -> bool {
    return current_count < 1000u;
}

// Attempt to acquire both physics and cell slots atomically
// Returns (physics_slot_idx, cell_slot_idx) on success, or (-1, -1) on failure
fn try_acquire_slots() -> vec2<i32> {
    let physics_free_count = atomicLoad(&points_free_list[0]);
    if physics_free_count == 0u {
        return vec2<i32>(-1, -1);
    }

    let cell_free_count = atomicLoad(&cells_free_list[0]);
    if cell_free_count == 0u {
        return vec2<i32>(-1, -1);
    }

    // Atomically decrement both free counts
    let new_physics_free_count = atomicSub(&points_free_list[0], 1u);
    let new_cell_free_count = atomicSub(&cells_free_list[0], 1u);

    if new_physics_free_count == 0u || new_cell_free_count == 0u {
        // Restore counts if we couldn't get both slots
        atomicAdd(&points_free_list[0], 1u);
        atomicAdd(&cells_free_list[0], 1u);
        return vec2<i32>(-1, -1);
    }

    // Get the actual slot indices
    let physics_slot_idx = atomicLoad(&points_free_list[new_physics_free_count]);
    let cell_slot_idx = atomicLoad(&cells_free_list[new_cell_free_count]);

    return vec2<i32>(i32(physics_slot_idx), i32(cell_slot_idx));
}

// Generate a random position within simulation bounds with padding
fn generate_random_position(seed: f32) -> vec2<f32> {
    let bounds_left = uniforms.bounds.x;
    let bounds_top = uniforms.bounds.y;
    let bounds_right = uniforms.bounds.z;
    let bounds_bottom = uniforms.bounds.w;

    let rand_x = rand_01(seed);
    let rand_y = rand_01(seed + 777.0);

    // Position randomly within bounds, with some padding from edges
    let padding = 10.0;
    let pos_x = bounds_left + padding + rand_x * (bounds_right - bounds_left - 2.0 * padding);
    let pos_y = bounds_top + padding + rand_y * (bounds_bottom - bounds_top - 2.0 * padding);

    return vec2<f32>(pos_x, pos_y);
}

// Generate initial random velocity

// Create a VerletPoint with the given position and velocity
fn create_point(position: vec2<f32>, velocity: vec2<f32>, radius: f32) -> VerletPoint {
    var point: VerletPoint;
    point.pos = position;
    point.prev_pos = position - velocity * uniforms.sim_params.x; // Set prev_pos to create initial velocity
    point.accel = vec2<f32>(0.0, 0.0);
    point.radius = radius;
    point.flags = POINT_FLAG_ACTIVE;
    return point;
}

// Create a Cell that references the given physics point index
fn create_cell(point_idx: u32, seed: f32) -> Cell {
    var cell: Cell;
    cell.point_idx = point_idx;
    cell.lifeform_id = 0u; // No lifeform initially
    cell.generation = 0u;
    cell.flags = 1u;
    cell.energy = 100.0; // Starting energy
    cell.cell_wall_thickness = 0.05;
    // Generate noise offset within safe bounds to prevent edge sampling discontinuities
    // Cell radius is ~25px, buffer is 50px, so total margin needed is 75px from each edge
    let margin = 75.0;
    let max_pos = 400.0 - margin;

    cell.noise_texture_offset = vec2<f32>(
        margin + rand_01(seed + 1000.0) * (max_pos - margin),
        margin + rand_01(seed + 2000.0) * (max_pos - margin)
    );

    // Generate random noise permutations for smooth cell wall variation
    let frequency = rand_01(seed + 1000.0) * 3.0 + 0.01;
    //cell.noise_permutations = generate_noise20_circle(u32(seed * 1000.0), 3.0, frequency, 2);

    return cell;
}

// Create a new cell directly in the simulation
fn spawn_cell(seed: f32) {
    let event_idx = atomicAdd(&event_counter, 1u);
    if (event_idx >= 2000u) {
        return;
    }
    // Try to acquire slots for both physics point and cell
    let slots = try_acquire_slots();
    if slots.x < 0 || slots.y < 0 {
        // No slots available
        return;
    }

    let physics_slot_idx = u32(slots.x);
    let cell_slot_idx = u32(slots.y);

    // Generate position and create physics point
    let position = generate_random_position(seed);
    let velocity = vec2<f32>(0.0, 0.0); // Start with no velocity
    let radius = rand_01(seed + position.x * 1000.0 + position.y * 10.0) * 1.0 + 0.1;
    let point = create_point(position, velocity, radius);

    // Increment the lifeform ID counter
    let new_lifeform_id = atomicAdd(&lifeform_id, 1u);

    // Create cell
    var cell = create_cell(physics_slot_idx, seed);
    cell.lifeform_id = new_lifeform_id;

    // Store the point and cell in their respective buffers
    points[physics_slot_idx] = point;
    cells[cell_slot_idx] = cell;

    // Increment counters
    atomicAdd(&points_counter, 1u);
    atomicAdd(&cells_counter, 1u);

    // Send a simple event notification for the new lifeform
    var event: Event;
    event.event_type = 1u;
    event.parent_lifeform_id = 4294967295u;  // No parent for new lifeforms
    event.lifeform_id = new_lifeform_id;
    event._pad = 0u;
    event_buffer[event_idx] = event;
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let current_count = atomicLoad(&points_counter);

    // Early return if we've reached capacity
    if !should_spawn_more(current_count) {
        return;
    }

    // Spawn a new cell directly
    let seed = f32(current_count);
    spawn_cell(seed);
}
