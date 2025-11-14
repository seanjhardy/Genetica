@include src/gpu/wgsl/types.wgsl;
@include src/gpu/wgsl/constants.wgsl;
@include src/gpu/wgsl/utils/genetic_algorithm.wgsl;
@include src/gpu/wgsl/utils/compute_collisions.wgsl;
@include src/gpu/wgsl/utils/events.wgsl;

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;


@group(0) @binding(1)
var<storage, read_write> cells: array<Cell>;


@group(0) @binding(2)
var<storage, read_write> cell_free_list: CellFreeList;


@group(0) @binding(3)
var<storage, read_write> alive_counter: Counter;

@group(0) @binding(4)
var<storage, read_write> spawn_buffer: SpawnBuffer;

@group(0) @binding(5)
var<storage, read_write> nutrient_grid: NutrientGrid;

@group(0) @binding(6)
var<storage, read_write> links: array<Link>;

@group(0) @binding(7)
var<storage, read_write> link_free_list: FreeList;

@group(0) @binding(8)
var<storage, read_write> link_events: LinkEventBuffer;

@group(0) @binding(9)
var<storage, read_write> cell_events: CellEventBuffer;

@group(0) @binding(10)
var<storage, read_write> cell_bucket_heads: array<atomic<i32>>;

@group(0) @binding(11)
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
var<storage, read_write> lifeform_events: LifeformEventBuffer;

@group(0) @binding(22)
var<storage, read_write> species_events: SpeciesEventBuffer;

fn compute_cell_color(energy: f32) -> vec4<f32> {
    let energy_normalized = clamp(energy / 100.0, 0.0, 1.0);
    let brightness = 0.1 + energy_normalized * 0.9;
    let r = (1.0 - brightness) * 0.5;
    let g = brightness;
    let b = brightness;
    return vec4<f32>(r, g, b, 1.0);
}


fn rand(seed: vec2<u32>) -> f32 {
    var x = seed.x * 1664525u + 1013904223u;
    var y = seed.y * 22695477u + 1u;
    let n = x ^ y;
    return f32(n & 0x00FFFFFFu) / f32(0x01000000u);
}

fn run_compiled_grn(
    descriptor: GrnDescriptor,
    receptors: u32,
    units: u32,
    stride: u32,
    lifeform_slot: u32,
) {
    if true {
        return;
    }
    if units == 0u || stride == 0u {
        return;
    }
    if lifeform_slot >= arrayLength(&lifeforms) {
        return;
    }
    let base_unit = descriptor.unit_offset;
    let total_units = arrayLength(&grn_units);
    if base_unit >= total_units {
        return;
    }
    let available_units = total_units - base_unit;
    let actual_units = min(units, available_units);
    for (var unit_idx: u32 = 0u; unit_idx < actual_units; unit_idx = unit_idx + 1u) {
        let unit = grn_units[base_unit + unit_idx];
        var additive_sum: f32 = 0.0;
        var multiplicative_acc: f32 = 1.0;
        var has_multiplicative: bool = false;
        let limit = min(unit.input_count, MAX_GRN_INPUTS_PER_UNIT);
        for (var input_idx: u32 = 0u; input_idx < limit; input_idx = input_idx + 1u) {
            let input = unit.inputs[input_idx];
            if input.index >= stride {
                continue;
            }
            let value = lifeforms[lifeform_slot].grn_state[input.index];
            let weighted = value * input.weight;
            if input.promoter_type == 1u {
                additive_sum = additive_sum + weighted;
            } else {
                has_multiplicative = true;
                let multiplicative_component = max(1.0 + weighted, 0.0);
                multiplicative_acc = multiplicative_acc * multiplicative_component;
            }
        }
        var output = additive_sum;
        if has_multiplicative {
            output = output + (multiplicative_acc - 1.0);
        }
        let target_index = receptors + unit_idx;
        if target_index < MAX_GRN_STATE_SIZE {
            lifeforms[lifeform_slot].grn_state[target_index] = output;
        }
    }
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let dt = uniforms.sim_params.x;

    let random = get_random_values(index);

    // Optimized: distribute population maintenance across all threads instead of just thread 0
    let total_threads = arrayLength(&cells);
    ensure_minimum_population_parallel(index, total_threads);

    spawn_cells();

    let total_cells = arrayLength(&cells);
    if index >= total_cells {
        return;
    }

    var cell = cells[index];
    if cell.is_alive == 0u {
        return;
    }

    // Update GRN: simple timer countdown, run when timer reaches 0

    // Decrease energy over time (metabolic rate)
    var energy_change_rate = 0.0;
    energy_change_rate += 0.2 + 0.3 / cell.radius; // Metabolism proportional to size
    energy_change_rate += 1000.0 * absorb_nutrients(index, 0.001 * cell.radius * cell.radius); // Eat nutrients from the environemnt
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

    let collision_correction = compute_collision_correction(index, cell.pos, cell.radius);
    if (collision_correction.x != 0.0) || (collision_correction.y != 0.0) {
        cell.pos += collision_correction;
        cell.prev_pos += collision_correction;
    }
    
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

    if cell.energy > MIN_DIVISION_ENERGY && random.w < DIVISION_PROBABILITY && cell.lifeform_slot < LIFEFORM_CAPACITY {
        let original_energy = cell.energy;
        let child_energy = original_energy * 0.5;
        let success = create_division_offspring(index, cell, child_energy);
        if success {
            cell.energy = child_energy;
            push_cell_event(
                CELL_EVENT_KIND_DIVISION,
                index,
                cell.lifeform_slot,
                0u,
                cell.pos,
                cell.radius,
                child_energy,
            );
        } else {
            cell.energy = original_energy;
        }
    }

    cell.color = compute_cell_color(cell.energy);
    cells[index] = cell;
}


fn spawn_cells() {
    loop {
        let prev_requests = atomicLoad(&spawn_buffer.counter.value);
        if prev_requests == 0u {
            break;
        }
        let desired_requests = prev_requests - 1u;
        let request_exchange = atomicCompareExchangeWeak(
            &spawn_buffer.counter.value,
            prev_requests,
            desired_requests,
        );
        if request_exchange.old_value == prev_requests && request_exchange.exchanged {
            let spawn_idx = desired_requests;
            var new_cell = spawn_buffer.requests[spawn_idx];
            let parent_marker = new_cell.metadata;
            new_cell.color = compute_cell_color(new_cell.energy);
            var spawned = false;
            loop {
                let free_prev = atomicLoad(&cell_free_list.count);
                if free_prev == 0u {
                    atomicAdd(&spawn_buffer.counter.value, 1u);
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
                    let previous_generation = cells[slot_index].metadata;
                    new_cell.metadata = previous_generation;
                    new_cell.is_alive = 1u;
                    cells[slot_index] = new_cell;
                    atomicAdd(&alive_counter.value, 1u);
                    let lf_idx = new_cell.lifeform_slot;
                    if lf_idx < LIFEFORM_CAPACITY {
                        if lf_idx < arrayLength(&lifeforms) {
                            let previous = lifeforms[lf_idx].cell_count;
                            lifeforms[lf_idx].cell_count = previous + 1u;
                            if previous == 0u {
                                lifeforms[lf_idx].first_cell_slot = slot_index;
                            }
                        }
                    }
                    if parent_marker != 0u {
                        let parent_index = parent_marker - 1u;
                        if parent_index < arrayLength(&cells) {
                            var link_created = false;
                            loop {
                                let link_prev = atomicLoad(&link_free_list.count);
                                if link_prev == 0u {
                                    break;
                                }
                                let link_desired = link_prev - 1u;
                                let link_exchange = atomicCompareExchangeWeak(
                                    &link_free_list.count,
                                    link_prev,
                                    link_desired,
                                );
                                if link_exchange.old_value == link_prev && link_exchange.exchanged {
                                    if link_desired < arrayLength(&link_free_list.indices) {
                                        let link_slot = link_free_list.indices[link_desired];
                                        let parent_cell = cells[parent_index];
                                        let rest_length = parent_cell.radius + new_cell.radius;
                                        links[link_slot].a = parent_index;
                                        links[link_slot].b = slot_index;
                                        links[link_slot].flags = LINK_FLAG_ALIVE | LINK_FLAG_ADHESIVE;
                                        links[link_slot].generation_a = parent_cell.metadata;
                                        links[link_slot].rest_length = rest_length;
                                        links[link_slot].stiffness = 0.6;
                                        links[link_slot].energy_transfer_rate = 0.0;
                                        links[link_slot].generation_b = new_cell.metadata;
                                        link_created = true;
                                        break;
                                    }
                                    atomicAdd(&link_free_list.count, 1u);
                                }
                            }
                            if !link_created {
                                let event_index = atomicAdd(&link_events.counter.value, 1u);
                                if event_index < arrayLength(&link_events.events) {
                                    link_events.events[event_index] = LinkEvent(
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
                                    atomicSub(&link_events.counter.value, 1u);
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

    cell.metadata = cell.metadata + 1u;
    cell.energy = 0.0;
    cell.is_alive = 0u;
    cell.color = compute_cell_color(cell.energy);
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
        if lf_idx < arrayLength(&lifeforms) {
            let previous = lifeforms[lf_idx].cell_count;
            if previous > 0u {
                let updated = previous - 1u;
                lifeforms[lf_idx].cell_count = updated;
                // Release lifeform if no cells remain and not preserved
                if updated == 0u && (lifeforms[lf_idx].flags & LIFEFORM_FLAG_PRESERVED) == 0u {
                    release_lifeform(lf_idx);
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

@compute @workgroup_size(128)
fn reset_bucket_heads(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let bucket_count = arrayLength(&cell_bucket_heads);
    if bucket_count == 0u || index >= bucket_count {
        return;
    }
    atomicStore(&cell_bucket_heads[index], -1);
}

@compute @workgroup_size(128)
fn build_spatial_hash(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let cell_count = arrayLength(&cells);
    if index >= cell_count {
        return;
    }

    let next_length = arrayLength(&cell_hash_next);
    if index < next_length {
        cell_hash_next[index] = -1;
    }

    let cell = cells[index];
    if cell.is_alive == 0u {
        return;
    }

    let bucket_index = hash_cell_position(cell.pos);
    let bucket_count = arrayLength(&cell_bucket_heads);
    if bucket_count == 0u || bucket_index >= bucket_count {
        return;
    }

    let previous = atomicExchange(&cell_bucket_heads[bucket_index], i32(index));
    if index < next_length {
        cell_hash_next[index] = previous;
    }
}

