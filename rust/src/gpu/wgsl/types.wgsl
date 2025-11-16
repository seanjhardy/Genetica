@include src/gpu/wgsl/constants.wgsl;

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
    generation: u32,
    parent_index: u32,
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
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

struct Lifeform {
    lifeform_id: u32,
    species_slot: u32,
    species_id: u32,
    gene_count: u32,
    rng_state: u32,
    first_cell_slot: u32,
    cell_count: atomic<u32>,
    grn_descriptor_slot: u32,
    grn_unit_offset: u32,
    grn_timer: u32,
    flags: u32,
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

// Genome structure: 200 genes, each with 55 base pairs (packed into 4 u32 words)
// - gene_ids[0..199]: Array of gene IDs (which genes are present, 0 = empty slot)
// - gene_sequences[0..799]: Packed base pair data
//   Each gene uses 4 consecutive u32 words (16 base pairs per word, 55 bases = 4 words)
//   Gene at index i uses words: gene_sequences[i*4 .. i*4+3]
//   Each base pair is 2 bits (0-3), packed into u32 words
struct GenomeEntry {
    gene_ids: array<u32, MAX_GENES_PER_GENOME>,
    gene_sequences: array<u32, GENOME_WORD_COUNT>,
}

struct SpeciesEntry {
    species_id: u32,
    mascot_lifeform_slot: u32,
    member_count: atomic<u32>,
    flags: u32,
    mascot_genome: GenomeEntry,
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

struct NutrientGrid {
    values: array<atomic<u32>>,
}

struct PositionChangeEntry {
    // Fixed-point: multiply by POSITION_CHANGE_SCALE for actual values
    delta_x: atomic<u32>,
    delta_y: atomic<u32>,
    num_changes: atomic<u32>,
    _pad: u32,
}


// Uniforms struct must match Rust struct layout exactly (including padding)
struct Uniforms {
    sim_params: vec4<f32>, // x: dt, y: zoom, z: view_w, w: view_h
    cell_count: vec4<f32>, // x: cell_count, y: reserved0, z: reserved1, w: reserved2
    camera: vec4<f32>,     // x: cam_x, y: cam_y
    bounds: vec4<f32>,     // (left, top, right, bottom)
    nutrient: vec4<u32>,// (Cell size, scale, reserved, reserved)
}

struct CellFreeList {
    count: atomic<u32>,
    indices: array<u32>,
}

struct Counter {
    value: atomic<u32>,
}
