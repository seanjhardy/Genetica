@include src/gpu/wgsl/constants.wgsl;
@include src/gpu/wgsl/types.wgsl;

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
/*se_link(idx: u32) {
    if idx >= arrayLength(&links) {
        return;
    }
    links[idx].flags = 0u;
    let slot = atomicAdd(&link_free_list.count, 1u);
    if slot < arrayLength(&link_free_list.indices) {
        link_free_list.indices[slot] = idx;
    } else {
        atomicSub(&link_free_list.count, 1u);
    }
}*/

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    /*let index = global_id.x;
    if index >= arrayLength(&links) {
        return;
    }

    let link = links[index];
    if (link.flags & LINK_FLAG_ACTIVE) == 0u {
        return;
    }

    if link.a_cell >= arrayLength(&cells) || link.b_cell >= arrayLength(&cells) {
        release_link(index);
        return;
    }

    if link.a_cell == link.b_cell {
        release_link(index);
        return;
    }

    var cell_a = cells[link.a_cell];
    var cell_b = cells[link.b_cell];

    if link.a_generation != cell_a.generation || link.b_generation != cell_b.generation {
        release_link(index);
        return;
    }

    if (cell_a.flags & CELL_FLAG_ACTIVE == 0u) || (cell_b.flags & CELL_FLAG_ACTIVE == 0u) {
        release_link(index);
        return;
    }

    if cell_a.point_idx >= arrayLength(&points) || cell_b.point_idx >= arrayLength(&points) {
        release_link(index);
        return;
    }

    let pa = points[cell_a.point_idx];
    let pb = points[cell_b.point_idx];

    let delta = pb.pos - pa.pos;
    var dist_sq = dot(delta, delta);
    if dist_sq <= 0.0001 {
        dist_sq = 0.0001;
    }

    let dist = sqrt(dist_sq);
    let max_distance = (cell_a.radius + cell_b.radius) * 2.0;
    if dist > max_distance {
        release_link(index);
        return;
    }

    let rest_length = link.rest_length;
    if rest_length == 0.0 {
        return;
    }

    // Simple Verlet distance constraint
    let diff = dist - rest_length;
    
    // Use stiffness to control how much of the correction is applied per frame
    // This prevents overshooting and oscillation
    let stiffness = clamp(link.stiffness, 0.0, 1.0);
    let correction = diff * stiffness;

    let a_weight = cell_a.radius * cell_a.radius;
    let b_weight = cell_b.radius * cell_b.radius;
    let inverse_a_weight = 1.0 / a_weight;
    let inverse_b_weight = 1.0 / b_weight;
    let inverse_total_weight = 1.0 / (a_weight + b_weight);

    // Normalize delta to get direction vector
    let delta_normalized = delta / dist;

    // Calculate position adjustments weighted by cell masses (area)
    // Apply stiffness to prevent over-correction and oscillation
    let adjustment_a_magnitude = correction * inverse_a_weight * inverse_total_weight;
    let adjustment_b_magnitude = correction * inverse_b_weight * inverse_total_weight;

    // Accumulate position changes in the buffer instead of modifying cells directly
    // This allows averaging multiple link forces and prevents race conditions
    if link.a_cell < arrayLength(&position_changes) && link.a_cell < arrayLength(&cells) {
        let adjustment_a: vec2<i32> = vec2<i32>(delta_normalized * adjustment_a_magnitude * POSITION_CHANGE_SCALE);
        atomicAdd(&position_changes[link.a_cell].delta_x, adjustment_a.x);
        atomicAdd(&position_changes[link.a_cell].delta_y, adjustment_a.y);
        atomicAdd(&position_changes[link.a_cell].num_changes, 1u);
    }

    if link.b_cell < arrayLength(&position_changes) && link.b_cell < arrayLength(&cells) {
        let adjustment_b: vec2<i32> = vec2<i32>(-delta_normalized * adjustment_b_magnitude * POSITION_CHANGE_SCALE);
        atomicAdd(&position_changes[link.b_cell].delta_x, adjustment_b.x);
        atomicAdd(&position_changes[link.b_cell].delta_y, adjustment_b.y);
        atomicAdd(&position_changes[link.b_cell].num_changes, 1u);
    }*/

    /*let energy_transfer_rate = link.energy_transfer_rate;
    let energy_difference = cell_a.energy - cell_b.energy;
    let energy_transfer = clamp(-energy_transfer_rate, energy_difference, energy_transfer_rate);
    cell_a.energy -= energy_transfer;
    cell_b.energy += energy_transfer;

    // Re-read cells right before writing to prevent overwriting changes made by other kernels
    // This is especially important for the color field which is computed by the cells kernel
    // If we don't re-read, we'll overwrite the color with stale data
    if link.a < arrayLength(&cells) && cell_a.is_alive != 0u {
        var latest_cell_a = cells[link.a];
        // Only update energy - preserve all other fields (especially color) from the latest read
        latest_cell_a.energy = cell_a.energy;
        cells[link.a] = latest_cell_a;
    }
    if link.b < arrayLength(&cells) && cell_b.is_alive != 0u {
        var latest_cell_b = cells[link.b];
        // Only update energy - preserve all other fields (especially color) from the latest read
        latest_cell_b.energy = cell_b.energy;
        cells[link.b] = latest_cell_b;
    }*/
}

