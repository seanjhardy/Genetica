@include src/gpu/wgsl/types.wgsl;
@include src/gpu/wgsl/constants.wgsl;
@include src/gpu/wgsl/utils/color.wgsl;
@include src/gpu/wgsl/utils/random.wgsl;
@include src/gpu/wgsl/utils/spawn_helpers.wgsl;

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
@group(0) @binding(10) var<storage, read_write> links: array<Link>;
@group(0) @binding(11) var<storage, read_write> link_free_list: array<atomic<u32>>;
@group(0) @binding(12) var<storage, read_write> division_requests: array<DivisionRequest>;
@group(0) @binding(13) var<storage, read_write> division_counter: atomic<u32>;

// Check if we should spawn more cells (below capacity limit)
fn should_spawn_more(current_count: u32) -> bool {
    return current_count < MIN_POINTS;
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

    // Randomize position/angle for initial spawn.
    let velocity = vec2<f32>(0.0, 0.0);
    let radius = rand_01(seed + 1000.0) * 5.0 + 0.1;
    let point_angle = rand_01(seed + 3.0) * (M_PI * 2.0);
    let position = generate_random_position(seed);

    let hue = 0.4 + rand_11(seed + 2000.0) * 0.05;
    let saturation = rand_01(seed + 4000.0) * 0.4 + 0.6;
    let lightness = rand_01(seed + 5000.0) * 0.1 + 0.2;

    let color = hsl_to_rgb(vec4<f32>(hue, saturation, lightness, 1.0));


    let new_lifeform_id = atomicAdd(&lifeform_id, 1u);

    var empty_parent: Cell;
    spawn_cell_at_slot(
        physics_slot_idx,
        cell_slot_idx,
        position,
        velocity,
        radius,
        color,
        point_angle,
        seed,
        false,
        empty_parent,
        new_lifeform_id,
        0u,
        100.0
    );

    // Increment counters
    atomicAdd(&points_counter, 1u);
    atomicAdd(&cells_counter, 1u);

    // Send a simple event notification for the new lifeform
    var event: Event;
    event.event_type = 1u;
    event.parent_lifeform_id = 0u;  // No parent for new lifeforms
    event.lifeform_id = new_lifeform_id;
    event._pad = 0u;
    event_buffer[event_idx] = event;
}

fn process_division_request(request: DivisionRequest) {
    // Safety check: ensure parent indices are valid
    if request.parent_cell_idx >= arrayLength(&cells) {
        return;
    }

    // Load parent cell data
    let parent_cell = cells[request.parent_cell_idx];
    if parent_cell.point_idx >= arrayLength(&points) {
        return;
    }

    let parent_point = points[parent_cell.point_idx];

    // Calculate daughter position
    let world_angle = parent_point.angle + request.angle;
    let offset = vec2<f32>(cos(world_angle), sin(world_angle)) * (parent_point.radius * 2.0);
    var daughter_pos = parent_point.pos + offset;

    // Clamp to bounds
    let min_bound = uniforms.bounds.xy + vec2<f32>(parent_point.radius, parent_point.radius);
    let max_bound = uniforms.bounds.zw - vec2<f32>(parent_point.radius, parent_point.radius);
    daughter_pos.x = clamp(daughter_pos.x, min_bound.x, max_bound.x);
    daughter_pos.y = clamp(daughter_pos.y, min_bound.y, max_bound.y);

    // Allocate slots for daughter (physics and cell only; no links)
    let slots = try_acquire_slots();
    if slots.x < 0 || slots.y < 0 {
        return; // No slots available
    }

    let point_slot_idx = u32(slots.x);
    let cell_slot_idx = u32(slots.y);

    // Safety check: ensure allocated slots are within buffer bounds
    if point_slot_idx >= arrayLength(&points) ||
       cell_slot_idx >= arrayLength(&cells) {
        // Note: try_acquire_slots already decremented the free lists.
        // In a production system, we'd need to restore them, but for now we'll skip.
        return;
    }

    // Create daughter cell
    spawn_cell_at_slot(
        point_slot_idx,
        cell_slot_idx,
        daughter_pos,
        vec2<f32>(0.0, 0.0),
        parent_point.radius,
        parent_cell.color,
        parent_point.angle,
        f32(request.parent_cell_idx), // Use parent index as seed
        true, // has_parent
        parent_cell,
        parent_cell.lifeform_id,
        request.generation,
        request.energy
    );

    // Increment counters for the new cells and link created
    atomicAdd(&points_counter, 1u);
    atomicAdd(&cells_counter, 1u);

    // Send event notification
    let event_idx = atomicAdd(&event_counter, 1u);
    if (event_idx < 2000u) {
        var event: Event;
        event.event_type = 2u; // Division event
        event.parent_lifeform_id = parent_cell.lifeform_id;
        event.lifeform_id = parent_cell.lifeform_id; // Same lifeform, new cell
        event._pad = request.generation;
        event_buffer[event_idx] = event;
    }
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Always process all pending division requests
    let division_count = atomicLoad(&division_counter);
    for (var i = 0u; i < division_count; i = i + 1u) {
        let request = division_requests[i];
        process_division_request(request);
    }

    // Reset division counter for next frame
    atomicStore(&division_counter, 0u);

    // Then spawn initial cells if needed (legacy behavior)
    let current_count = atomicLoad(&points_counter);
    if should_spawn_more(current_count) {
        let seed = f32(current_count);
        spawn_cell(seed);
    }
}
