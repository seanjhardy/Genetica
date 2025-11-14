@include src/gpu/wgsl/types.wgsl;
@include src/gpu/wgsl/constants.wgsl;

fn push_lifeform_event(kind: u32, lifeform_id: u32, species_id: u32, lifeform_slot: u32) {
    let event_index = atomicAdd(&lifeform_events.counter.value, 1u);
    if event_index < arrayLength(&lifeform_events.events) {
        lifeform_events.events[event_index] = LifeformEvent(
            kind,
            lifeform_id,
            species_id,
            lifeform_slot,
        );
    } else {
        atomicSub(&lifeform_events.counter.value, 1u);
    }
}

fn push_species_event(kind: u32, species_id: u32, species_slot: u32, member_count: u32) {
    let event_index = atomicAdd(&species_events.counter.value, 1u);
    if event_index < arrayLength(&species_events.events) {
        species_events.events[event_index] = SpeciesEvent(
            kind,
            species_id,
            species_slot,
            member_count,
        );
    } else {
        atomicSub(&species_events.counter.value, 1u);
    }
}

fn allocate_lifeform_slot() -> u32 {
    loop {
        let prev = atomicLoad(&lifeform_free.count);
        if prev == 0u {
            break;
        }
        let desired = prev - 1u;
        let exchange = atomicCompareExchangeWeak(&lifeform_free.count, prev, desired);
        if exchange.old_value == prev && exchange.exchanged {
            if desired < arrayLength(&lifeform_free.indices) {
                return lifeform_free.indices[desired];
            }
            break;
        }
    }
    return LIFEFORM_CAPACITY;
}

fn recycle_lifeform_slot(slot: u32) {
    let index = atomicAdd(&lifeform_free.count, 1u);
    if index < arrayLength(&lifeform_free.indices) {
        lifeform_free.indices[index] = slot;
    }
}

fn allocate_species_slot() -> u32 {
    loop {
        let prev = atomicLoad(&species_free.count);
        if prev == 0u {
            break;
        }
        let desired = prev - 1u;
        let exchange = atomicCompareExchangeWeak(&species_free.count, prev, desired);
        if exchange.old_value == prev && exchange.exchanged {
            if desired < arrayLength(&species_free.indices) {
                return species_free.indices[desired];
            }
            break;
        }
    }
    return MAX_SPECIES_CAPACITY;
}

fn recycle_species_slot(slot: u32) {
    let index = atomicAdd(&species_free.count, 1u);
    if index < arrayLength(&species_free.indices) {
        species_free.indices[index] = slot;
    }
}

fn queue_spawn_cell(new_cell: Cell) -> bool {
    let index = atomicAdd(&spawn_buffer.counter.value, 1u);
    if index < arrayLength(&spawn_buffer.requests) {
        spawn_buffer.requests[index] = new_cell;
        return true;
    }
    atomicSub(&spawn_buffer.counter.value, 1u);
    return false;
}

fn read_genome_base(slot: u32, base_index: u32) -> u32 {
    if slot >= arrayLength(&genomes) {
        return 0u;
    }
    let word_index = base_index / 16u;
    let offset = (base_index & 15u) * 2u;
    if word_index >= GENOME_WORD_COUNT {
        return 0u;
    }
    let word = genomes[slot].base_pairs[word_index];
    return (word >> offset) & 0x3u;
}

fn write_genome_base(slot: u32, base_index: u32, value: u32) {
    if slot >= arrayLength(&genomes) {
        return;
    }
    let word_index = base_index / 16u;
    let offset = (base_index & 15u) * 2u;
    if word_index >= GENOME_WORD_COUNT {
        return;
    }
    let mask = ~(0x3u << offset);
    let word = genomes[slot].base_pairs[word_index];
    genomes[slot].base_pairs[word_index] = (word & mask) | ((value & 0x3u) << offset);
}

fn copy_genome(dest_slot: u32, src_slot: u32) {
    if dest_slot >= arrayLength(&genomes) || src_slot >= arrayLength(&genomes) {
        return;
    }
    genomes[dest_slot].gene_count = genomes[src_slot].gene_count;
    for (var i: u32 = 0u; i < GENOME_WORD_COUNT; i = i + 1u) {
        genomes[dest_slot].base_pairs[i] = genomes[src_slot].base_pairs[i];
    }
}

fn random_genome(slot: u32, seed: u32) {
    if slot >= arrayLength(&genomes) {
        return;
    }
    genomes[slot].gene_count = MAX_GENES_PER_GENOME;
    for (var base_index: u32 = 0u; base_index < BASE_PAIRS_PER_GENOME; base_index = base_index + 1u) {
        let random_value = rand(vec2<u32>(seed + base_index * 17u, slot * 97u + base_index * 13u));
        let base = u32(random_value * 4.0);
        write_genome_base(slot, base_index, base);
    }
}

fn mutate_genome(child_slot: u32, parent_slot: u32, seed: u32) -> u32 {
    var differences: u32 = 0u;
    for (var base_index: u32 = 0u; base_index < BASE_PAIRS_PER_GENOME; base_index = base_index + 1u) {
        let random_value = rand(vec2<u32>(seed + base_index * 31u, child_slot * 131u + base_index));
        if random_value < MUTATE_BASE_CHANCE {
            let new_base = u32(rand(vec2<u32>(seed * 3u + base_index * 7u, child_slot * 211u)) * 4.0);
            var parent_base: u32;
            if parent_slot < LIFEFORM_CAPACITY {
                parent_base = read_genome_base(parent_slot, base_index);
            } else {
                parent_base = read_genome_base(child_slot, base_index);
            }
            if new_base != parent_base {
                differences = differences + 1u;
            }
            write_genome_base(child_slot, base_index, new_base);
        }
    }
    return differences;
}

fn create_species(child_slot: u32) -> vec2<u32> {
    let slot = allocate_species_slot();
    if slot >= MAX_SPECIES_CAPACITY {
        return vec2<u32>(MAX_SPECIES_CAPACITY, 0u);
    }
    let species_id = atomicAdd(&next_species_id.value, 1u);
    if slot < arrayLength(&species_entries) {
        species_entries[slot].species_id = species_id;
        species_entries[slot].mascot_lifeform_slot = child_slot;
        atomicStore(&species_entries[slot].member_count, 1u);
        species_entries[slot].flags = SPECIES_FLAG_ACTIVE;
    }
    push_species_event(SPECIES_EVENT_KIND_CREATE, species_id, slot, 1u);
    return vec2<u32>(slot, species_id);
}

fn release_species(slot: u32, species_id: u32) {
    if slot >= arrayLength(&species_entries) {
        return;
    }
    species_entries[slot].species_id = 0u;
    species_entries[slot].mascot_lifeform_slot = 0u;
    atomicStore(&species_entries[slot].member_count, 0u);
    species_entries[slot].flags = 0u;
    recycle_species_slot(slot);
    push_species_event(SPECIES_EVENT_KIND_EXTINCT, species_id, slot, 0u);
}

fn assign_species(child_slot: u32, parent_slot: u32, differences: u32) -> vec2<u32> {
    if parent_slot < LIFEFORM_CAPACITY {
        let parent_entry = lifeforms[parent_slot];
        if (parent_entry.flags & LIFEFORM_FLAG_ACTIVE) != 0u {
            if differences >= SPECIES_DIFF_THRESHOLD {
                return create_species(child_slot);
            } else {
                let species_slot = parent_entry.species_slot;
                if species_slot < arrayLength(&species_entries) {
                    atomicAdd(&species_entries[species_slot].member_count, 1u);
                    species_entries[species_slot].flags = SPECIES_FLAG_ACTIVE;
                    return vec2<u32>(species_slot, parent_entry.species_id);
                }
            }
        }
    }
    return create_species(child_slot);
}

fn release_lifeform(lifeform_slot: u32) {
    if lifeform_slot >= LIFEFORM_CAPACITY {
        return;
    }
    let entry = lifeforms[lifeform_slot];
    if (entry.flags & LIFEFORM_FLAG_ACTIVE) == 0u {
        return;
    }

    let species_slot = entry.species_slot;
    if species_slot < arrayLength(&species_entries) {
        let previous = atomicSub(&species_entries[species_slot].member_count, 1u);
        if previous <= 1u {
            release_species(species_slot, entry.species_id);
        }
    }

    if lifeform_slot < arrayLength(&lifeforms) {
        lifeforms[lifeform_slot].lifeform_id = 0u;
        lifeforms[lifeform_slot].species_slot = 0u;
        lifeforms[lifeform_slot].species_id = 0u;
        lifeforms[lifeform_slot].gene_count = 0u;
        lifeforms[lifeform_slot].rng_state = 0u;
        lifeforms[lifeform_slot].cell_count = 0u;
        lifeforms[lifeform_slot].flags = 0u;
        lifeforms[lifeform_slot].first_cell_slot = 0u;
        lifeforms[lifeform_slot].grn_descriptor_slot = 0u;
        lifeforms[lifeform_slot].grn_unit_offset = 0u;
        lifeforms[lifeform_slot].grn_timer = 0u;
        lifeforms[lifeform_slot]._pad = 0u;
        lifeforms[lifeform_slot]._pad2 = vec2<u32>(0u, 0u);
        for (var i: u32 = 0u; i < MAX_GRN_STATE_SIZE; i = i + 1u) {
            lifeforms[lifeform_slot].grn_state[i] = 0.0;
        }
    }

    recycle_lifeform_slot(lifeform_slot);
    push_lifeform_event(LIFEFORM_EVENT_KIND_DESTROY, entry.lifeform_id, entry.species_id, lifeform_slot);
}

fn random_position(seed: vec2<u32>) -> vec2<f32> {
    let width = uniforms.bounds.z - uniforms.bounds.x;
    let height = uniforms.bounds.w - uniforms.bounds.y;
    let rx = rand(seed);
    let ry = rand(seed.yx);
    return vec2<f32>(
        uniforms.bounds.x + rx * width,
        uniforms.bounds.y + ry * height,
    );
}

fn initialise_lifeform_state(slot: u32, lifeform_id: u32) {
    if slot >= arrayLength(&lifeforms) {
        return;
    }
    lifeforms[slot].lifeform_id = lifeform_id;
    lifeforms[slot].first_cell_slot = 0u;
    lifeforms[slot].cell_count = 0u;
    lifeforms[slot].grn_descriptor_slot = slot;
    lifeforms[slot].grn_unit_offset = slot * MAX_GRN_REGULATORY_UNITS;
    lifeforms[slot].flags = LIFEFORM_FLAG_ACTIVE;
    lifeforms[slot].grn_timer = 0u;
    lifeforms[slot]._pad = 0u;
    lifeforms[slot]._pad2 = vec2<u32>(0u, 0u);
    for (var i: u32 = 0u; i < MAX_GRN_STATE_SIZE; i = i + 1u) {
        lifeforms[slot].grn_state[i] = 0.0;
    }
}

fn initialise_grn(slot: u32) {
    if slot >= arrayLength(&grn_descriptors) {
        return;
    }
    grn_descriptors[slot].receptor_count = 0u;
    grn_descriptors[slot].unit_count = 0u;
    grn_descriptors[slot].state_stride = 0u;
    grn_descriptors[slot].unit_offset = slot * MAX_GRN_REGULATORY_UNITS;
}

fn create_lifeform_cell(
    parent_slot: u32,
    position: vec2<f32>,
    radius: f32,
    energy: f32,
    metadata: u32,
    seed: vec2<u32>,
) -> bool {
    let lifeform_slot = allocate_lifeform_slot();
    if lifeform_slot >= LIFEFORM_CAPACITY {
        return false;
    }

    let lifeform_id = atomicAdd(&next_lifeform_id.value, 1u);

    if parent_slot < LIFEFORM_CAPACITY && (lifeforms[parent_slot].flags & LIFEFORM_FLAG_ACTIVE) != 0u {
        lifeforms[lifeform_slot].gene_count = lifeforms[parent_slot].gene_count;
        copy_genome(lifeform_slot, parent_slot);
    } else {
        lifeforms[lifeform_slot].gene_count = MAX_GENES_PER_GENOME;
        random_genome(lifeform_slot, lifeform_id + 1u);
    }

    let differences = mutate_genome(lifeform_slot, parent_slot, lifeform_id + 11u);
    let species_info = assign_species(lifeform_slot, parent_slot, differences);
    let species_slot = species_info.x;
    let species_id = species_info.y;

    lifeforms[lifeform_slot].lifeform_id = lifeform_id;
    let valid_species = species_slot < MAX_SPECIES_CAPACITY;
    if valid_species {
        lifeforms[lifeform_slot].species_slot = species_slot;
        lifeforms[lifeform_slot].species_id = species_id;
    } else {
        lifeforms[lifeform_slot].species_slot = 0u;
        lifeforms[lifeform_slot].species_id = 0u;
    }
    lifeforms[lifeform_slot].rng_state = lifeform_id * 1664525u + 1013904223u;
    lifeforms[lifeform_slot].cell_count = 0u;
    lifeforms[lifeform_slot].flags = LIFEFORM_FLAG_ACTIVE;
    lifeforms[lifeform_slot]._pad = 0u;

    initialise_lifeform_state(lifeform_slot, lifeform_id);
    initialise_grn(lifeform_slot);

    push_lifeform_event(LIFEFORM_EVENT_KIND_CREATE, lifeform_id, species_id, lifeform_slot);

    var new_cell: Cell;
    new_cell.pos = position;
    new_cell.prev_pos = position;
    new_cell.random_force = vec2<f32>(0.0, 0.0);
    new_cell.radius = radius;
    new_cell.energy = energy;
    new_cell.cell_wall_thickness = 0.1;
    new_cell.is_alive = 1u;
    new_cell.lifeform_slot = lifeform_slot;
    new_cell.metadata = metadata;
    new_cell.color = compute_cell_color(energy);

    if !queue_spawn_cell(new_cell) {
        release_lifeform(lifeform_slot);
        return false;
    }

    return true;
}

fn create_random_lifeform(seed: u32) -> bool {
    let position = random_position(vec2<u32>(seed * 97u + 11u, seed * 131u + 23u));
    let radius = 1.0 + rand(vec2<u32>(seed * 17u + 7u, seed * 29u + 3u)) * 3.0;
    let energy = 60.0 + rand(vec2<u32>(seed * 53u + 5u, seed * 71u + 19u)) * 80.0;
    return create_lifeform_cell(
        LIFEFORM_CAPACITY,
        position,
        radius,
        energy,
        0u,
        vec2<u32>(seed * 191u + 37u, seed * 223u + 41u),
    );
}

// Optimized: distribute population maintenance across threads instead of running serially on thread 0
// Each thread checks if it should create a lifeform based on its index
fn ensure_minimum_population_parallel(thread_index: u32, total_threads: u32) {
    let alive = atomicLoad(&alive_counter.value);
    if alive >= MIN_ACTIVE_CELLS {
        return;
    }
    let deficit = MIN_ACTIVE_CELLS - alive;
    
    // Distribute deficit across threads - each thread handles its portion
    let lifeforms_per_thread = (deficit + total_threads - 1u) / total_threads;
    let start_index = thread_index * lifeforms_per_thread;
    let end_index = min(start_index + lifeforms_per_thread, deficit);

    for (var i: u32 = start_index; i < end_index; i = i + 1u) {
        // Check again if we've reached minimum (another thread might have created lifeforms)
        let current_alive = atomicLoad(&alive_counter.value);
        if current_alive >= MIN_ACTIVE_CELLS {
            break;
        }
        if !create_random_lifeform(i + current_alive * 13u + 1u) {
            break;
        }
    }
}

fn create_division_offspring(parent_index: u32, parent_cell: Cell, child_energy: f32) -> bool {
    let offset_seed = vec2<u32>(parent_index * 97u + 13u, parent_cell.lifeform_slot * 211u + 17u);
    let angle = rand(offset_seed) * 6.2831853;
    let distance = parent_cell.radius * 0.75 + 0.5;
    let child_position = parent_cell.pos + vec2<f32>(cos(angle), sin(angle)) * distance;
    let child_radius = parent_cell.radius;
    return create_lifeform_cell(
        parent_cell.lifeform_slot,
        child_position,
        child_radius,
        child_energy,
        parent_index + 1u,
        offset_seed,
    );
}
