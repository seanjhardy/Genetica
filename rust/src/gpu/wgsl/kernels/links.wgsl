@include src/gpu/wgsl/constants.wgsl;
@include src/gpu/wgsl/types.wgsl;
@include src/gpu/wgsl/utils/events.wgsl;

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(0) @binding(1)
var<storage, read_write> cells: array<Cell>;

@group(0) @binding(2)
var<storage, read_write> cell_free_list: CellFreeList;

@group(0) @binding(3)
var<storage, read_write> cell_counter: Counter;

@group(0) @binding(4)
var<storage, read_write> spawn_buffer: SpawnBuffer;

@group(0) @binding(5)
var<storage, read_write> nutrient_grid: NutrientGrid;

@group(0) @binding(6)
var<storage, read_write> links: array<Link>;

@group(0) @binding(7)
var<storage, read_write> link_free_list: FreeList;

@group(0) @binding(8)
var<storage, read_write> cell_bucket_heads: array<atomic<i32>>;

@group(0) @binding(9)
var<storage, read_write> cell_hash_next: array<i32>;

@group(0) @binding(12)
var<storage, read_write> grn_descriptors: array<GrnDescriptor>;

@group(0) @binding(13)
var<storage, read> grn_units: array<CompiledRegulatoryUnit>;

@group(0) @binding(14)
var<storage, read_write> lifeforms: array<Lifeform>;

@group(0) @binding(15)
var<storage, read_write> lifeform_free: FreeList;

@group(0) @binding(16)
var<storage, read_write> next_lifeform_id: Counter;

@group(0) @binding(17)
var<storage, read_write> genomes: array<GenomeEntry>;

@group(0) @binding(18)
var<storage, read_write> species_entries: array<SpeciesEntry>;

@group(0) @binding(19)
var<storage, read_write> species_free: FreeList;

@group(0) @binding(20)
var<storage, read_write> next_species_id: Counter;

@group(0) @binding(21)
var<storage, read_write> position_changes: array<PositionChangeEntry>;

fn compute_cell_color(energy: f32) -> vec4<f32> {
    let energy_normalized = clamp(energy / 100.0, 0.0, 1.0);
    let brightness = 0.1 + energy_normalized * 0.9;
    let r = (1.0 - brightness) * 0.5;
    let g = brightness;
    let b = brightness;
    return vec4<f32>(r, g, b, 1.0);
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if index >= arrayLength(&links) {
        return;
    }

    let link = links[index];
    if (link.flags & LINK_FLAG_ALIVE) == 0u {
        return;
    }

    if link.a >= arrayLength(&cells) || link.b >= arrayLength(&cells) {
        release_link(index);
        return;
    }

    var cell_a = cells[link.a];
    var cell_b = cells[link.b];

    if link.generation_a != cell_a.metadata || link.generation_b != cell_b.metadata {
        release_link(index);
        return;
    }

    if cell_a.is_alive == 0u || cell_b.is_alive == 0u {
        release_link(index);
        return;
    }

    let delta = cell_b.pos - cell_a.pos;
    let dist_sq = dot(delta, delta);
    if dist_sq == 0.0 {
        return;
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

    let stiffness = link.stiffness;
    let diff = dist - rest_length;
    let correction = (diff / dist) * 0.5 * stiffness;
    let adjustment = delta * correction;

    // Accumulate position changes in the buffer instead of modifying cells directly
    // This allows averaging multiple link forces and prevents race conditions
    if link.a < arrayLength(&position_changes) {
        let adjustment_x_fixed = i32(adjustment.x * POSITION_CHANGE_SCALE);
        let adjustment_y_fixed = i32(adjustment.y * POSITION_CHANGE_SCALE);
        atomicAdd(&position_changes[link.a].delta_x, u32(adjustment_x_fixed));
        atomicAdd(&position_changes[link.a].delta_y, u32(adjustment_y_fixed));
        atomicAdd(&position_changes[link.a].num_changes, 1u);
    }

    if link.b < arrayLength(&position_changes) {
        let adjustment_x_fixed = i32(-adjustment.x * POSITION_CHANGE_SCALE);
        let adjustment_y_fixed = i32(-adjustment.y * POSITION_CHANGE_SCALE);
        atomicAdd(&position_changes[link.b].delta_x, u32(adjustment_x_fixed));
        atomicAdd(&position_changes[link.b].delta_y, u32(adjustment_y_fixed));
        atomicAdd(&position_changes[link.b].num_changes, 1u);
    }

    let energy_transfer_rate = link.energy_transfer_rate;
    let energy_difference = cell_a.energy - cell_b.energy;
    let energy_transfer = clamp(-energy_transfer_rate, energy_difference, energy_transfer_rate);
    cell_a.energy -= energy_transfer;
    cell_b.energy += energy_transfer;

    cell_a.color = compute_cell_color(cell_a.energy);
    cell_b.color = compute_cell_color(cell_b.energy);

    cells[link.a] = cell_a;
    cells[link.b] = cell_b;
}

