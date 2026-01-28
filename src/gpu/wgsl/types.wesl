import package::constants::{
    MAX_CELL_LINKS,
    CELL_WALL_SAMPLES,
    MAX_GENES_PER_GENOME,
    GENOME_WORD_COUNT,
    MAX_GRN_INPUTS_PER_UNIT,
};

struct Point {
    pos: vec2<f32>,
    prev_pos: vec2<f32>,
    accel: vec2<f32>,
    angle: f32,
    radius: f32,
    flags: u32,
}

struct Cell {
    point_idx: u32,
    lifeform_id: u32,
    generation: u32,
    energy: f32,
    cell_wall_thickness: f32,
    _pad0: array<u32, 3>,
    color: vec4<f32>,
    flags: u32,
    link_count: u32,
    link_indices: array<u32, MAX_CELL_LINKS>,
    noise_permutations: array<f32, CELL_WALL_SAMPLES>,
    noise_texture_offset: vec2<f32>,
    //inputs: array<f32, CELL_INPUT_ARRAY_SIZE>,
}

struct Link {
    a_cell: u32,
    a_generation: u32,
    b_cell: u32,
    b_generation: u32,
    angle_from_a: f32,
    angle_from_b: f32,
    stiffness: f32,
    flags: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

struct Lifeform {
    lifeform_id: u32,
    species_id: u32,
    first_cell: u32,
    cell_count: atomic<u32>,
    flags: u32,
    _pad: vec3<u32>,
}

struct Species {
    species_id: u32,
    ancestor_species_id: u32,
    member_count: atomic<u32>,
    flags: u32,
}

struct CellFreeList {
    count: atomic<u32>,
    _pad: u32,
    indices: array<u32>,
}

struct SpeciesEntry {
    species_id: u32,
    ancestor_species_id: u32,
    member_count: atomic<u32>,
    flags: u32,
}

struct GenomeEntry {
    gene_ids: array<u32, MAX_GENES_PER_GENOME>,
    gene_sequences: array<u32, GENOME_WORD_COUNT>,
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
    unit_count: u32,
    unit_start_index: u32,
    _pad0: u32,
    _pad1: u32,
}

struct FreeList {
    count: atomic<u32>,
    indices: array<u32>,
}

struct NutrientGrid {
    values: array<atomic<u32>>,
}

struct Uniforms {
    sim_params: vec4<f32>, // x: dt, y: zoom, z: view_w, w: view_h
    cell_count: vec4<f32>, // x: cell_count, y: reserved0, z: reserved1, w: reserved2
    camera: vec4<f32>,     // x: cam_x, y: cam_y
    bounds: vec4<f32>,     // (left, top, right, bottom)
    nutrient: vec4<u32>,   // (Cell size, scale, reserved, reserved)
    selection: vec4<u32>,  // (selected_cell, reserved0, reserved1, reserved2)
    seed: vec4<u32>,       // (spawn_seed, reserved0, reserved1, reserved2)
}


struct DivisionRequest {
    parent_cell_idx: u32,
    generation: u32,
    energy: f32,
    angle: f32,
}

struct Event {
    event_type: u32,
    parent_lifeform_id: u32,
    lifeform_id: u32,
    _pad: u32,
}
