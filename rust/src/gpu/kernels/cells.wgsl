const MAX_GRN_RECEPTOR_INPUTS: u32 = 16u;
const MAX_GRN_REGULATORY_UNITS: u32 = 16u;
const MAX_GRN_INPUTS_PER_UNIT: u32 = 8u;
const MAX_GRN_STATE_SIZE: u32 = MAX_GRN_RECEPTOR_INPUTS + MAX_GRN_REGULATORY_UNITS;
const MAX_GENES_PER_GENOME: u32 = 200u;
const BASE_PAIRS_PER_GENE: u32 = 20u;
const BASE_PAIRS_PER_GENOME: u32 = MAX_GENES_PER_GENOME * BASE_PAIRS_PER_GENE;
const GENOME_WORD_COUNT: u32 = (BASE_PAIRS_PER_GENOME + 3u) / 4u;
const GRN_EVALUATION_FALLBACK: u32 = 8u;
const LIFEFORM_STATE_FLAG_ACTIVE: u32 = 1u;
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
    color: vec4<f32>,
}

struct GrnInput {
    weight: f32,
    index: u32,
    promoter_type: u32,
    _pad: u32,
}

struct CompiledRegulatoryUnit {
    input_count: u32,
    output_index: u32,
    flags: u32,
    _padding: u32,
    inputs: array<GrnInput, MAX_GRN_INPUTS_PER_UNIT>,
}

struct GrnDescriptor {
    receptor_count: u32,
    unit_count: u32,
    state_stride: u32,
    unit_offset: u32,
    evaluation_interval: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct LifeformState {
    lifeform_id: u32,
    first_cell_slot: u32,
    cell_count: u32,
    grn_descriptor_slot: u32,
    grn_unit_offset: u32,
    grn_unit_count: u32,
    flags: u32,
    grn_receptor_count: u32,
    grn_timer: u32,
    _pad: u32,
    _pad2: vec2<u32>,
    grn_state: array<f32, MAX_GRN_STATE_SIZE>,
}

struct Link {
    a: u32,
    b: u32,
    flags: u32,
    generation_a: u32,
    rest_length: f32,
    stiffness: f32,
    energy_transfer_rate: f32,
    generation_b: u32,
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

struct GenomeEntry {
    gene_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    base_pairs: array<u32, GENOME_WORD_COUNT>,
}

struct LifeformEntry {
    lifeform_id: u32,
    species_slot: u32,
    species_id: u32,
    gene_count: u32,
    rng_state: u32,
    cell_count: u32,
    flags: u32,
    _pad: u32,
}

struct SpeciesEntry {
    species_id: u32,
    mascot_lifeform_slot: u32,
    member_count: atomic<u32>,
    flags: u32,
}

struct LifeformEvent {
    kind: u32,
    lifeform_id: u32,
    species_id: u32,
    lifeform_slot: u32,
}

struct SpeciesEvent {
    kind: u32,
    species_id: u32,
    species_slot: u32,
    member_count: u32,
}

struct FreeList {
    count: atomic<u32>,
    _pad: u32,
    indices: array<u32>,
}

struct SpawnBuffer {
    counter: Counter,
    _pad: u32,
    requests: array<Cell>,
}

struct CellEventBuffer {
    counter: Counter,
    _pad: u32,
    events: array<CellEvent>,
}

struct LinkEventBuffer {
    counter: Counter,
    _pad: u32,
    events: array<LinkEvent>,
}

struct LifeformEventBuffer {
    counter: Counter,
    _pad: u32,
    events: array<LifeformEvent>,
}

struct SpeciesEventBuffer {
    counter: Counter,
    _pad: u32,
    events: array<SpeciesEvent>,
}

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
        let parent_entry = lifeform_entries[parent_slot];
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
    let entry = lifeform_entries[lifeform_slot];
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

    atomicStore(&lifeform_active.values[lifeform_slot], 0u);

    lifeform_entries[lifeform_slot].lifeform_id = 0u;
    lifeform_entries[lifeform_slot].species_slot = 0u;
    lifeform_entries[lifeform_slot].species_id = 0u;
    lifeform_entries[lifeform_slot].gene_count = 0u;
    lifeform_entries[lifeform_slot].rng_state = 0u;
    lifeform_entries[lifeform_slot].cell_count = 0u;
    lifeform_entries[lifeform_slot].flags = 0u;
    lifeform_entries[lifeform_slot]._pad = 0u;

    if lifeform_slot < arrayLength(&lifeform_states) {
        lifeform_states[lifeform_slot].flags = 0u;
        lifeform_states[lifeform_slot].lifeform_id = 0u;
        lifeform_states[lifeform_slot].cell_count = 0u;
        lifeform_states[lifeform_slot].grn_descriptor_slot = 0u;
        lifeform_states[lifeform_slot].grn_unit_count = 0u;
        lifeform_states[lifeform_slot].grn_unit_offset = 0u;
        lifeform_states[lifeform_slot].grn_receptor_count = 0u;
        lifeform_states[lifeform_slot].grn_timer = 0u;
        lifeform_states[lifeform_slot]._pad = 0u;
        lifeform_states[lifeform_slot]._pad2 = vec2<u32>(0u, 0u);
        for (var i: u32 = 0u; i < MAX_GRN_STATE_SIZE; i = i + 1u) {
            lifeform_states[lifeform_slot].grn_state[i] = 0.0;
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
    if slot >= arrayLength(&lifeform_states) {
        return;
    }
    lifeform_states[slot].lifeform_id = lifeform_id;
    lifeform_states[slot].first_cell_slot = 0u;
    lifeform_states[slot].cell_count = 0u;
    lifeform_states[slot].grn_descriptor_slot = slot;
    lifeform_states[slot].grn_unit_offset = slot * MAX_GRN_REGULATORY_UNITS;
    lifeform_states[slot].grn_unit_count = 0u;
    lifeform_states[slot].flags = LIFEFORM_STATE_FLAG_ACTIVE;
    lifeform_states[slot].grn_receptor_count = 0u;
    lifeform_states[slot].grn_timer = 0u;
    lifeform_states[slot]._pad = 0u;
    lifeform_states[slot]._pad2 = vec2<u32>(0u, 0u);
    for (var i: u32 = 0u; i < MAX_GRN_STATE_SIZE; i = i + 1u) {
        lifeform_states[slot].grn_state[i] = 0.0;
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
    grn_descriptors[slot].evaluation_interval = GRN_EVALUATION_FALLBACK;
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

    if parent_slot < LIFEFORM_CAPACITY && (lifeform_entries[parent_slot].flags & LIFEFORM_FLAG_ACTIVE) != 0u {
        lifeform_entries[lifeform_slot].gene_count = lifeform_entries[parent_slot].gene_count;
        copy_genome(lifeform_slot, parent_slot);
    } else {
        lifeform_entries[lifeform_slot].gene_count = MAX_GENES_PER_GENOME;
        random_genome(lifeform_slot, lifeform_id + 1u);
    }

    let differences = mutate_genome(lifeform_slot, parent_slot, lifeform_id + 11u);
    let species_info = assign_species(lifeform_slot, parent_slot, differences);
    let species_slot = species_info.x;
    let species_id = species_info.y;

    lifeform_entries[lifeform_slot].lifeform_id = lifeform_id;
    let valid_species = species_slot < MAX_SPECIES_CAPACITY;
    if valid_species {
        lifeform_entries[lifeform_slot].species_slot = species_slot;
        lifeform_entries[lifeform_slot].species_id = species_id;
    } else {
        lifeform_entries[lifeform_slot].species_slot = 0u;
        lifeform_entries[lifeform_slot].species_id = 0u;
    }
    lifeform_entries[lifeform_slot].rng_state = lifeform_id * 1664525u + 1013904223u;
    lifeform_entries[lifeform_slot].cell_count = 0u;
    lifeform_entries[lifeform_slot].flags = LIFEFORM_FLAG_ACTIVE;
    lifeform_entries[lifeform_slot]._pad = 0u;

    initialise_lifeform_state(lifeform_slot, lifeform_id);
    initialise_grn(lifeform_slot);

    atomicStore(&lifeform_active.values[lifeform_slot], 0u);
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

fn ensure_minimum_population() {
    let alive = atomicLoad(&alive_counter.value);
    if alive >= MIN_ACTIVE_CELLS {
        return;
    }
    let deficit = MIN_ACTIVE_CELLS - alive;
    for (var i: u32 = 0u; i < deficit; i = i + 1u) {
        if !create_random_lifeform(i + alive * 13u + 1u) {
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
var<storage, read_write> spawn_buffer: SpawnBuffer;

struct LifeformFlagArray {
    values: array<atomic<u32>>,
}

@group(0) @binding(5)
var<storage, read_write> lifeform_active: LifeformFlagArray;

struct NutrientGrid {
    values: array<atomic<u32>>,
}

@group(0) @binding(6)
var<storage, read_write> nutrient_grid: NutrientGrid;

@group(0) @binding(7)
var<storage, read_write> links: array<Link>;

@group(0) @binding(8)
var<storage, read_write> link_free_list: FreeList;

@group(0) @binding(9)
var<storage, read_write> link_events: LinkEventBuffer;

@group(0) @binding(10)
var<storage, read_write> cell_events: CellEventBuffer;

@group(0) @binding(11)
var<storage, read_write> cell_bucket_heads: array<atomic<i32>>;

@group(0) @binding(12)
var<storage, read_write> cell_hash_next: array<i32>;

@group(0) @binding(13)
var<storage, read_write> grn_descriptors: array<GrnDescriptor>;

@group(0) @binding(14)
var<storage, read> grn_units: array<CompiledRegulatoryUnit>;

@group(0) @binding(15)
var<storage, read_write> lifeform_states: array<LifeformState>;

@group(0) @binding(16)
var<storage, read_write> lifeform_entries: array<LifeformEntry>;

@group(0) @binding(17)
var<storage, read_write> lifeform_free: FreeList;

@group(0) @binding(18)
var<storage, read_write> next_lifeform_id: Counter;

@group(0) @binding(19)
var<storage, read_write> genomes: array<GenomeEntry>;

@group(0) @binding(20)
var<storage, read_write> species_entries: array<SpeciesEntry>;

@group(0) @binding(21)
var<storage, read_write> species_free: FreeList;

@group(0) @binding(22)
var<storage, read_write> next_species_id: Counter;

@group(0) @binding(23)
var<storage, read_write> lifeform_events: LifeformEventBuffer;

@group(0) @binding(24)
var<storage, read_write> species_events: SpeciesEventBuffer;

const DIVISION_PROBABILITY: f32 = 0.0001;
const RANDOM_DEATH_PROBABILITY: f32 = 0.00005;
const LIFEFORM_CAPACITY: u32 = 10000u;
const MIN_DIVISION_ENERGY: f32 = 20.0;

const CELL_EVENT_KIND_DIVISION: u32 = 1u;
const CELL_EVENT_KIND_DEATH: u32 = 2u;
const CELL_EVENT_FLAG_ADHESIVE: u32 = 1u;

const LINK_EVENT_KIND_CREATE: u32 = 1u;
const LINK_EVENT_KIND_REMOVE: u32 = 2u;

const LINK_FLAG_ALIVE: u32 = 1u;
const LINK_FLAG_ADHESIVE: u32 = 1u << 1u;

const LIFEFORM_FLAG_ACTIVE: u32 = 1u;
const SPECIES_FLAG_ACTIVE: u32 = 1u;

const LIFEFORM_EVENT_KIND_CREATE: u32 = 1u;
const LIFEFORM_EVENT_KIND_DESTROY: u32 = 2u;

const SPECIES_EVENT_KIND_CREATE: u32 = 1u;
const SPECIES_EVENT_KIND_EXTINCT: u32 = 2u;

const MIN_ACTIVE_CELLS: u32 = 20u;
const MUTATE_BASE_CHANCE: f32 = 0.0005;
const SPECIES_DIFF_THRESHOLD: u32 = 200u;
const MAX_SPECIES_CAPACITY: u32 = 1024u;

const HASH_CELL_SIZE: f32 = 8.0;
const COLLISION_EPSILON: f32 = 0.0001;

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
    if units == 0u || stride == 0u {
        return;
    }
    if lifeform_slot >= arrayLength(&lifeform_states) {
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
            let value = lifeform_states[lifeform_slot].grn_state[input.index];
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
            lifeform_states[lifeform_slot].grn_state[target_index] = output;
        }
    }
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
    let event_index = atomicAdd(&cell_events.counter.value, 1u);
    if event_index < arrayLength(&cell_events.events) {
        cell_events.events[event_index] = CellEvent(
            kind,
            parent_cell_index,
            parent_lifeform_slot,
            flags,
            position,
            radius,
            energy,
        );
    } else {
        atomicSub(&cell_events.counter.value, 1u);
    }
}

fn hash_cell_position(pos: vec2<f32>) -> u32 {
    let bucket_count = arrayLength(&cell_bucket_heads);
    if bucket_count == 0u {
        return 0u;
    }
    let grid = vec2<i32>(floor(pos / HASH_CELL_SIZE));
    let hashed = (grid.x * 73856093) ^ (grid.y * 19349663);
    let mask = bucket_count - 1u;
    return u32(hashed) & mask;
}

fn compute_collision_correction(index: u32, position: vec2<f32>, radius: f32) -> vec2<f32> {
    let bucket_count = arrayLength(&cell_bucket_heads);
    if bucket_count == 0u {
        return vec2<f32>(0.0, 0.0);
    }

    let cell_capacity = arrayLength(&cells);
    let next_length = arrayLength(&cell_hash_next);

    var correction = vec2<f32>(0.0, 0.0);

    var dx: i32 = -1;
    loop {
        if dx > 1 {
            break;
        }

        var dy: i32 = -1;
        loop {
            if dy > 1 {
                break;
            }

            let neighbor_pos = position + vec2<f32>(f32(dx), f32(dy)) * HASH_CELL_SIZE;
            let neighbor_hash = hash_cell_position(neighbor_pos);

            var head = atomicLoad(&cell_bucket_heads[neighbor_hash]);
            loop {
                if head == -1 {
                    break;
                }

                let neighbor_index = u32(head);
                if neighbor_index != index && neighbor_index < cell_capacity {
                    let neighbor = cells[neighbor_index];
                    if neighbor.is_alive != 0u {
                        let delta = position - neighbor.pos;
                        let dist_sq = dot(delta, delta);
                        let min_dist = radius + neighbor.radius;
                        if min_dist > 0.0 && dist_sq < (min_dist * min_dist) {
                            let dist = sqrt(max(dist_sq, COLLISION_EPSILON));
                            var push_dir = vec2<f32>(0.0, 0.0);
                            if dist > 0.0 {
                                push_dir = delta / dist;
                            }
                            if push_dir.x == 0.0 && push_dir.y == 0.0 {
                                if index < neighbor_index {
                                    push_dir = vec2<f32>(1.0, 0.0);
                                } else {
                                    push_dir = vec2<f32>(-1.0, 0.0);
                                }
                            }
                            let overlap = min_dist - dist;
                            if overlap > 0.0 {
                                correction += push_dir * (overlap * 0.5);
                            }
                        }
                    }
                }

                var next_head: i32 = -1;
                if neighbor_index < next_length {
                    next_head = cell_hash_next[neighbor_index];
                }
                head = next_head;
            }

            dy = dy + 1;
        }

        dx = dx + 1;
    }

    return correction;
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let dt = uniforms.sim_params.x;

    let random = get_random_values(index);

    if index == 0u {
        ensure_minimum_population();
    }

    spawn_cells();

    let total_cells = arrayLength(&cells);
    if index >= total_cells {
        return;
    }

    var cell = cells[index];
    if cell.is_alive == 0u {
        return;
    }

    if cell.lifeform_slot < LIFEFORM_CAPACITY && cell.lifeform_slot < arrayLength(&lifeform_states) {
        let lf_index = cell.lifeform_slot;
        let descriptor_slot = lifeform_states[lf_index].grn_descriptor_slot;
        if (lifeform_states[lf_index].flags & LIFEFORM_STATE_FLAG_ACTIVE) == 0u
            || descriptor_slot >= arrayLength(&grn_descriptors) {
            lifeform_states[lf_index].grn_receptor_count = 0u;
            lifeform_states[lf_index].grn_unit_count = 0u;
            lifeform_states[lf_index].grn_timer = 0u;
        } else {
            let descriptor = grn_descriptors[descriptor_slot];
            let receptors = min(descriptor.receptor_count, MAX_GRN_RECEPTOR_INPUTS);
            let units = min(descriptor.unit_count, MAX_GRN_REGULATORY_UNITS);
            let stride = min(descriptor.state_stride, MAX_GRN_STATE_SIZE);
            if stride == 0u || units == 0u {
                lifeform_states[lf_index].grn_receptor_count = receptors;
                lifeform_states[lf_index].grn_unit_count = units;
                lifeform_states[lf_index].grn_timer = 0u;
            } else {
                if lifeform_states[lf_index].grn_receptor_count != receptors
                    || lifeform_states[lf_index].grn_unit_count != units {
                    lifeform_states[lf_index].grn_receptor_count = receptors;
                    lifeform_states[lf_index].grn_unit_count = units;
                    for (var i: u32 = 0u; i < MAX_GRN_STATE_SIZE; i = i + 1u) {
                        lifeform_states[lf_index].grn_state[i] = 0.0;
                    }
                    lifeform_states[lf_index].grn_timer = 0u;
                }
                if lifeform_states[lf_index].grn_timer == 0u {
                    let interval = select(descriptor.evaluation_interval, GRN_EVALUATION_FALLBACK, descriptor.evaluation_interval == 0u);
                    run_compiled_grn(descriptor, receptors, units, stride, lf_index);
                    lifeform_states[lf_index].grn_timer = max(interval, 1u);
                } else {
                    lifeform_states[lf_index].grn_timer = lifeform_states[lf_index].grn_timer - 1u;
                }
            }
        }
    }

    // Decrease energy over time (metabolic rate)
    var energy_change_rate = 0.0;
    energy_change_rate -= 0.2 + 0.3 / cell.radius; // Metabolism proportional to size
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
                        let previous = atomicAdd(&lifeform_active.values[lf_idx], 1u);
                        if lf_idx < arrayLength(&lifeform_entries) {
                            lifeform_entries[lf_idx].cell_count = previous + 1u;
                        }
                        if lf_idx < arrayLength(&lifeform_states) {
                            lifeform_states[lf_idx].cell_count = previous + 1u;
                            if previous == 0u {
                                lifeform_states[lf_idx].first_cell_slot = slot_index;
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
                                        let rest_length =
                                            parent_cell.radius + new_cell.radius;
                                        links[link_slot].a = parent_index;
                                        links[link_slot].b = slot_index;
                                        links[link_slot].flags =
                                            LINK_FLAG_ALIVE | LINK_FLAG_ADHESIVE;
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
                                let event_index =
                                    atomicAdd(&link_events.counter.value, 1u);
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
        let previous = atomicLoad(&lifeform_active.values[lf_idx]);
        if previous > 0u {
            let old_value = atomicSub(&lifeform_active.values[lf_idx], 1u);
            var updated: u32;
            if old_value > 0u {
                updated = old_value - 1u;
            } else {
                updated = 0u;
            }
            if lf_idx < arrayLength(&lifeform_entries) {
                lifeform_entries[lf_idx].cell_count = updated;
            }
            if lf_idx < arrayLength(&lifeform_states) {
                lifeform_states[lf_idx].cell_count = updated;
            }
            if old_value == 1u {
                atomicStore(&lifeform_active.values[lf_idx], 0u);
                release_lifeform(lf_idx);
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

